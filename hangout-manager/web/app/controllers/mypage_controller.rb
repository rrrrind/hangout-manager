class MypageController < ApplicationController
  before_action :set_user
  def show
  end

  def follow
  end

  def follower
  end
  
  private
    # Use callbacks to share common setup or constraints between actions.
    def set_user
      @user = User.find(params[:id])
    end
end

