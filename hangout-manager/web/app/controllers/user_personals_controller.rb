class UserPersonalsController < ApplicationController
  before_action :set_user_personal, only: [:show, :edit, :update, :destroy]

  # GET /user_personals
  # GET /user_personals.json
  def index
    @user_personals = UserPersonal.all
  end

  def get_json
    @user_personal = UserPersonal.find(params[:id])
    render json: @user_personal
  end

  def get_all_json
    @user_personals = UserPersonal.all
    @users = User.all
    render :json => { :users => @users, :user_personals => @user_personals }
  end

  # GET /user_personals/1
  # GET /user_personals/1.json
  def show
  end

  # GET /user_personals/new
  def new
    @user_personal = UserPersonal.new
  end

  # GET /user_personals/1/edit
  def edit
  end

  # POST /user_personals
  # POST /user_personals.json
  def create
    @user_personal = UserPersonal.new(user_personal_params)
    @user_personal.user_id = current_user.id
    @user_personal.save
    redirect_to new_user_detail_path
  end

  # PATCH/PUT /user_personals/1
  # PATCH/PUT /user_personals/1.json
  def update
    respond_to do |format|
      if @user_personal.update(user_personal_params)
        format.html { redirect_to @user_personal, notice: 'User personal was successfully updated.' }
        format.json { render :show, status: :ok, location: @user_personal }
      else
        format.html { render :edit }
        format.json { render json: @user_personal.errors, status: :unprocessable_entity }
      end
    end
  end

  # DELETE /user_personals/1
  # DELETE /user_personals/1.json
  def destroy
    @user_personal.destroy
    respond_to do |format|
      format.html { redirect_to user_personals_url, notice: 'User personal was successfully destroyed.' }
      format.json { head :no_content }
    end
  end

  private
    # Use callbacks to share common setup or constraints between actions.
    def set_user_personal
      @user_personal = UserPersonal.find(params[:id])
    end

    # Only allow a list of trusted parameters through.
    def user_personal_params
      params.require(:user_personal).permit(:p_one, :p_two, :p_three, :p_four, :user)
    end
end
