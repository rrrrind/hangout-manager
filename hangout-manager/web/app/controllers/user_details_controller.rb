class UserDetailsController < ApplicationController
  before_action :set_user_detail, only: [:show, :edit, :update, :destroy]

  # GET /user_details
  # GET /user_details.json
  def index
    @user_details = UserDetail.all
  end

  # GET /user_details/1
  # GET /user_details/1.json
  def show
  end

  def get_json
    @user_detail = UserDetail.find(params[:id])
    render json: @user_detail
  end

  # GET /user_details/new
  def new
    @user_detail = UserDetail.new
  end

  # GET /user_details/1/edit
  def edit
  end

  # POST /user_details
  # POST /user_details.json
  def create
    @user_detail = UserDetail.new(user_detail_params)
    @user_detail.user_id = current_user.id
    @user_detail.save
    redirect_to "http://localhost:3000/user/#{current_user.id}"
  end

  # PATCH/PUT /user_details/1
  # PATCH/PUT /user_details/1.json
  def update
    respond_to do |format|
      if @user_detail.update(user_detail_params)
        format.html { redirect_to @user_detail, notice: 'User detail was successfully updated.' }
        format.json { render :show, status: :ok, location: @user_detail }
      else
        format.html { render :edit }
        format.json { render json: @user_detail.errors, status: :unprocessable_entity }
      end
    end
  end

  # DELETE /user_details/1
  # DELETE /user_details/1.json
  def destroy
    @user_detail.destroy
    respond_to do |format|
      format.html { redirect_to user_details_url, notice: 'User detail was successfully destroyed.' }
      format.json { head :no_content }
    end
  end

  private
    # Use callbacks to share common setup or constraints between actions.
    def set_user_detail
      @user_detail = UserDetail.find(params[:id])
    end

    # Only allow a list of trusted parameters through.
    def user_detail_params
      params.require(:user_detail).permit(:d_one, :d_two, :d_three, :d_four, :d_five, :d_six, :d_seven, :d_eight, :user)
    end
end
